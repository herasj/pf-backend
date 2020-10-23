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
    return await this.tweetService.findRandom();
  }

  @ApiOperation({ summary: 'Update political' })
  @Put('random')
  async updatePolitical(@Body() data: UpdatePoliticalTweetDTO) {
    return await this.tweetService.updateOne(data);
  }
}
