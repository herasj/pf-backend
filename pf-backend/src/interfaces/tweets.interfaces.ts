import { ApiOperation, ApiProperty } from '@nestjs/swagger';
import { Document } from 'mongoose';
export interface ITweetsModel extends Document {
  tweetId: string;
  userId: string;
  createdAt: string;
  text: string;
  replies: number;
  retweets: number;
  favorites: number;
  hashtag: string;
  location: ILocation;
  url?: string;
  sentimentScore: ISentiment;
  political: boolean;
}

export interface ITweet {
  tweetId: string;
  userId: string;
  createdAt: string;
  text: string;
  replies: number;
  retweets: number;
  favorites: number;
  hashtag: string;
  location: ILocation;
  url?: string;
  sentimentScore: ISentiment;
  political: boolean;
}

export interface ISentiment {
  predominant: string;
  positive: number;
  negative: number;
  neutral: number;
  mixed: number;
}

export interface ILocation {
  latitude: number;
  longitude: number;
}

export class ITweetResponse {
  @ApiProperty({type: String})
  _id: string;
  
  @ApiProperty({type: String})
  tweetId: string;

  @ApiProperty({type: String})
  username: string;

  @ApiProperty({type: String})
  name: string;

  @ApiProperty({type: String})
  createdAt: string;
}