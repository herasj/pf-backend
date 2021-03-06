import { ApiOkResponse, ApiOperation, ApiResponse, ApiTags } from '@nestjs/swagger';
import { Body, Controller, Get, Param, Put, Query } from '@nestjs/common';
import { ITweetResponse } from '../../interfaces/tweets.interfaces';
import { UpdatePoliticalTweetDTO } from './dtos/tweet.dto';
import { TweetsService } from './tweets.service';

@ApiTags('Tweet')
@Controller('tweet')
export class TweetsController {
  constructor(private readonly tweetService: TweetsService) {}

  @ApiOperation({ summary: 'Get random tweet' })
  @Get('random')
  async getRandom() {
    const randomTweet = await this.tweetService.findRandom();
    return randomTweet[0];
  }

  @ApiOperation({ summary: 'Get random political tweet' })
  @Get('random/political')
  async getRandomPolitical() {
    const randomTweet = await this.tweetService.findRandomPolitical();
    return randomTweet[0];
  }

   @ApiOperation({summary: 'Get graph info (Latest 15 extraction dates)'})
   @Get('graph')
   async getGraphInfo(){
     return await this.tweetService.graphInfo()
   }

  @ApiOkResponse({type: [ITweetResponse], description: 'Returns an array of tweets'})
  @ApiOperation({ summary: 'Get latest tweets (Page size = 10)' })
  @Get('latest')
  async getLatest( @Query('page') page: number)   {
    if (!page || Number(page) < 1) page = 1;
    return await this.tweetService.getLatest(Number(page));
  }

  @ApiOperation({ summary: 'Get negative tweets' })
  @Get('negative')
  async getNegative() {
    return await this.tweetService.getNegativeTweets();
  }

  @ApiOperation({ summary: 'Get details of tweet' })
  @Get(':id/details')
  async getDetails(@Param('id') tweetId: string) {
    return await this.tweetService.getDetails(tweetId);
  }

  @ApiOkResponse({type: [ITweetResponse], description: 'Returns an array of tweets'})
  @ApiOperation({ summary: 'Get tweets from user (Page size = 10)' })
  @Get('user/:userId')
  async getUserTweets(
    @Param('userId') userId: string,
    @Query('page') page: number,
  ) {
    if (!page || Number(page) < 1) page = 1;
    return await this.tweetService.getUserTweets(userId, Number(page));
  }

  @ApiOperation({ summary: 'Get location' })
  @Get('location')
  async getLocation() {
    return await this.tweetService.getLocation();
  }

  @ApiOperation({ summary: 'Count tweets' })
  @Get('count')
  async countTweets() {
    return { count: await this.tweetService.countTweets() };
  }

  @ApiOperation({ summary: 'Count today tweets' })
  @Get('count/today')
  async getTodayTweets() {
    return { count: await this.tweetService.getTodayTweets() };
  }

  @ApiOperation({ summary: 'Count political tweets' })
  @Get('count/political')
  async countPoliticalTweets() {
    return { count: await this.tweetService.countPoliticalTweets() };
  }

  @ApiOperation({ summary: 'Update polarization' })
  @Put('random')
  async updatePolitical(@Body() data: UpdatePoliticalTweetDTO) {
    const polarization = data.political
    return await this.tweetService.updateOne(data._id, polarization);
  }
}
