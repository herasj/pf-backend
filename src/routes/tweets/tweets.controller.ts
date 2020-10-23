import { ApiOperation, ApiResponse, ApiTags } from '@nestjs/swagger';
import { Body, Controller, Get, Put } from '@nestjs/common';
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

  @ApiOperation({ summary: 'Get latest 25 tweets' })
  @Get('latest')
  async getLatest() {
    return await this.tweetService.getLatest();
  }

  @ApiOperation({ summary: 'Get negative tweets' })
  @Get('negative')
  async getNegative() {
    return await this.tweetService.getNegativeTweets();
  }

  @ApiOperation({ summary: 'Get location' })
  @Get('location')
  async getLocation() {
    return await this.tweetService.getLocation();
  }

  @ApiOperation({ summary: 'Count political tweets' })
  @Get('count/political')
  async countPoliticalTweets() {
    return { count: await this.tweetService.countPoliticalTweets() };
  }

  @ApiOperation({ summary: 'Count tweets' })
  @Get('count')
  async countTweets() {
    return { count: await this.tweetService.countTweets() };
  }

  @ApiOperation({ summary: 'Update political' })
  @Put('random')
  async updatePolitical(@Body() data: UpdatePoliticalTweetDTO) {
    return await this.tweetService.updateOne(data);
  }
}
